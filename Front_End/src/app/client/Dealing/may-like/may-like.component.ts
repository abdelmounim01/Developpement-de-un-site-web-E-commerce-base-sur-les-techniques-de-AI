import { Component, OnInit } from '@angular/core';
import { NgModule } from '@angular/core';
import { SlickCarouselModule } from 'ngx-slick-carousel';
import {ProjetService} from  'src/app/service/ImplementationService/projet.service' ; 
import {Task} from 'src/app/service/Interfaces/task'
import { Routes, Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { PriceService } from 'src/app/service/PriceService/price.service';

@Component({
  selector: 'app-may-like',
  templateUrl: './may-like.component.html',
  styleUrls: ['./may-like.component.css']
})
export class MayLikeComponent implements OnInit {
  tasks: Task[]; 
  iduser:number = JSON.parse(localStorage.getItem('user')).userId;
  /* if you change the nbr of slidesToShow you should change the size of with in card in css file */
  slideConfig = {"slidesToShow": 5, "slidesToScroll": 4};
  message:string;

  constructor(private price: PriceService  , private http: HttpClient , private resourceService:ProjetService , public router:Router) { }

  ngOnInit() {
    this.price.currentMessage.subscribe(message => this.message = message) ; 

    this.maylike(this.iduser)
  }

  maylike(iduser):void {
    this.resourceService.maylike(this.iduser).subscribe(tasks => (this.tasks = tasks))
  }
}
