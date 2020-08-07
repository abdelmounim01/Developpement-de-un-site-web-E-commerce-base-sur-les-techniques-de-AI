import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { NgModule } from '@angular/core';
import { SlickCarouselModule } from 'ngx-slick-carousel';
import {ProjetService} from  'src/app/service/ImplementationService/projet.service' ; 
import {Task} from 'src/app/service/Interfaces/task'
import { Routes, Router } from '@angular/router';
import { PriceService } from 'src/app/service/PriceService/price.service';



@Component({
  selector: 'app-most-popular',
  templateUrl: './most-popular.component.html',
  styleUrls: ['./most-popular.component.css']
})
export class MostPopularComponent implements OnInit {
  tasks: Task[]; 
  message:string;


  constructor(private price: PriceService  , private http: HttpClient , private resourceService:ProjetService , public router:Router) { }

  ngOnInit(){
    this.price.currentMessage.subscribe(message => this.message = message) ; 

    this.MostPopular() ; 
  }

  slides = [
    {img: "assets/images/Home/jacket.jpg"},
    {img: "assets/images/Home/phone.jpg"},
    {img: "assets/images/Home/sandal.jpg"},
    {img: "assets/images/Home/shirt.jpg"},
    {img: "assets/images/Home/shoes.jpg"},
    {img: "assets/images/Home/jacket.jpg"},
    {img: "assets/images/Home/jenz.jpg"},
    {img: "assets/images/Home/shoes.jpg"},
    {img: "assets/images/Home/phone.jpg"}
 
  ];
  /* if you change the nbr of slidesToShow you should change the size of with in card in css file */
  slideConfig = {"slidesToShow": 5, "slidesToScroll": 4};
  
  
  MostPopular (): void {
    this.resourceService.popularproducts().subscribe(tasks => (this.tasks = tasks))
  }

 }
