import { Component, OnInit } from '@angular/core'
import { HttpClient } from '@angular/common/http'
import { Metauser } from '../../../service/Interfaces/metauser'
import { ProjetService } from '../../../service/ImplementationService/projet.service'


@Component({
  selector: 'app-metausers',
  templateUrl: './metausers.component.html',
  styleUrls: ['./metausers.component.css']
})
export class MetausersComponent implements OnInit {
  metausers:Metauser[]
  p: number=1;
  searchmetauser: string
  constructor(private projetService: ProjetService, private http: HttpClient) { }

  ngOnInit(): void {
    this.getMetausers()
  }
  getMetausers (): void {
    this.projetService.getMetausers().subscribe(metausers => (this.metausers = metausers))
  }

}
