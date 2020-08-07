import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../../../service/ImplementationService/projet.service'
import { Metauser } from 'src/app/service/Interfaces/metauser';

@Component({
  selector: 'app-rated',
  templateUrl: './rated.component.html',
  styleUrls: ['./rated.component.css']
})
export class RatedComponent implements OnInit {
  metauser:Metauser[]
  metauserlike:Metauser[]
  metauserrating:Metauser[]
  Metauserpanier:Metauser[]
  UserId:number = JSON.parse(localStorage.getItem('user')).userId
  constructor(private projetService: ProjetService) { }

  ngOnInit(): void {
    this.getMetauser(this.UserId)
  }
  getMetauser (UserId): void {
    this.projetService.getMetauser(UserId).subscribe(metauser => (this.metauser = metauser))
  }
}
